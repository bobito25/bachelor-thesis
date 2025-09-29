

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(on b f)
(on c e)
(ontable d)
(on e d)
(on f a)
(clear b)
)
(:goal
(and
(on a f)
(on c a)
(on d b)
(on e d))
)
)


