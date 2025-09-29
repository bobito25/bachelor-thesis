

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b c)
(ontable c)
(on d e)
(on e a)
(on f d)
(clear b)
(clear f)
)
(:goal
(and
(on b e)
(on c d)
(on d f)
(on e c)
(on f a))
)
)


