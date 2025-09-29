

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b c)
(on c f)
(ontable d)
(on e d)
(on f e)
(clear a)
(clear b)
)
(:goal
(and
(on a d)
(on c e)
(on d c)
(on f b))
)
)


