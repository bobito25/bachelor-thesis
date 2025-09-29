

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b d)
(ontable c)
(on d a)
(on e c)
(on f e)
(clear b)
(clear f)
)
(:goal
(and
(on a c)
(on d b)
(on e d)
(on f a))
)
)


